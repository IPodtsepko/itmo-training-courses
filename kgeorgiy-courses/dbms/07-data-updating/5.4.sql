create view StudentDebts as
select
    StudentId,
    count(
        distinct case
            when Mark is null then CourseId
        end
    ) as Debts
from
    Students
    left join Plan using (GroupId)
    left join Marks using (StudentId, CourseId)
group by
    StudentId;
