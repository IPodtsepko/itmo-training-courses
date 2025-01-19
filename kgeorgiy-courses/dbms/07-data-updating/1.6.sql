delete from
    Students
where
    StudentId in (
        select
            StudentId
        from
            Students
            left join Plan using (GroupId)
            left join Marks using (StudentId, CourseId)
        group by
            StudentId
        having
            count(CourseId) > count(Mark)
    );
