-- Postgres 15.4

create function SameMarks() returns trigger as
$$
begin
	if not exists (
		select StudentId, CourseId
		from
			Students natural join (
				select distinct GroupId, CourseId
				from Students natural join Marks
			) WholeSetOfCoursesByGroups
		except
		select StudentId, CourseId from Marks
	) then
		return new;
	end if;
	raise exception 'Students from the same group have marks in different courses';
end;
$$ language plpgsql;

create trigger SameMarks
after insert or update or delete on Marks
execute function SameMarks();

create trigger SameMarks
after insert or update on Students
execute function SameMarks();
